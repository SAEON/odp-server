import copy
from datetime import date, datetime, timezone

from sqlalchemy import and_, select
from sqlalchemy.orm import aliased

from odp.const import ODPPackageTag, ODPDateRangeIncType
from odp.db import Session
from odp.db.models import PackageTag, Package
from odp.package import PackageModule


class DateRangeInc(PackageModule):

    def _internal_execute(self):
        """
        Fetch and increment date of package date range tag according to it's related date range increment tag.
        """
        date_range_package_tag: PackageTag = aliased(PackageTag)
        date_range_inc_package_tag: PackageTag = aliased(PackageTag)

        stmt = (
            select(Package, date_range_package_tag, date_range_inc_package_tag)
            .join(
                date_range_package_tag,
                and_(
                    Package.id == date_range_package_tag.package_id,
                    date_range_package_tag.tag_id == ODPPackageTag.DATERANGE
                )
            )
            .join(
                date_range_inc_package_tag,
                and_(
                    Package.id == date_range_inc_package_tag.package_id,
                    date_range_inc_package_tag.tag_id == ODPPackageTag.DATERANGEINC
                )
            )
        )

        for (package, date_range_package_tag, date_range_inc_package_tag) in Session.execute(stmt).all():
            date_range_data = self._get_updated_date_range_data(date_range_package_tag.data, date_range_inc_package_tag)
            self._update_date_range_package_tag(date_range_package_tag, date_range_data, package)

    @staticmethod
    def _get_updated_date_range_data(date_range_package_tag_data, date_range_inc_package_tag) -> dict:
        date_range_data = copy.deepcopy(date_range_package_tag_data)
        # Iterate through the date range increment types and update the corresponding date range dates.
        for (date_type, increment_type) in date_range_inc_package_tag.data.items():
            updated_date = date.today()

            match increment_type:
                case ODPDateRangeIncType.CURRENT_DATE:
                    updated_date = date.today().strftime("%Y/%m/%d")

            date_range_data[date_type] = updated_date

        return date_range_data

    @staticmethod
    def _update_date_range_package_tag(date_range_package_tag, updated_data, package):
        # Save the changes and update the timestamps.
        date_range_package_tag.data = updated_data
        date_range_package_tag.timestamp = (timestamp := datetime.now(timezone.utc))
        date_range_package_tag.save()

        package.timestamp = timestamp
        package.save()
